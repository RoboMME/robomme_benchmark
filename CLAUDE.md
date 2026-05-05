# CLAUDE.md — RoboMME Benchmark

## 与用户沟通规范

- **始终使用中文**与用户沟通，包括所有文字回复、计划说明、错误提示和进度汇报。
- **计划（Plan）必须用中文撰写**，包括 Context、Approach、Verification 等各节。
- **代码、命令、变量名、注释**保持英文原样，不翻译。
- **端到端验证优先**：完成任何代码改动后，必须通过实际运行脚本或测试来验证功能正确性，除非用户明确指示跳过或说明无法运行（如缺少 GPU/display）。轻量单元测试只能作为辅助，不能代替端到端验证。

---

## Project Overview

RoboMME is a robotic benchmark for memory-augmented manipulation (CVPR 2026 challenge). It wraps [ManiSkill](https://github.com/haosulab/ManiSkill) environments into 16 tasks across 4 task suites. Participants implement a `Policy` class, serve it via `challenge_interface/`, and the organizer evaluates it end-to-end.

Source code lives in `src/robomme/`. Tests live in `tests/`. Challenge integration code lives in `challenge_interface/`.

---

## Environment & Package Management (uv)

This repo uses **uv** exclusively. Never use bare `pip`, `python`, or `conda` unless explicitly asked.

```bash
# First-time setup
uv sync
uv pip install -e .

# With optional server dependencies (challenge interface)
uv sync --group server

# Add / remove a dependency
uv add <package>
uv remove <package>

# Always run commands through uv
uv run scripts/run_example.py
uv run python -m pytest tests/
```

Python version is pinned at **3.11** (`.python-version`). `pyproject.toml` is the single source of truth; `uv.lock` must stay consistent with it.

`mani-skill` is sourced from a specific Git commit — do not change `[tool.uv.sources]` without a reason.

### Dependency Groups

| Group | Purpose | Install command |
|-------|---------|-----------------|
| (default) | core sim + training | `uv sync` |
| `dev` | pytest + opencv dev tools | `uv sync --group dev` |
| `server` | Flask + msgpack + websockets (challenge interface) | `uv sync --group server` |

---

## Error Handling Rules

**No silent fallbacks.** Every error path must raise explicitly. No `except Exception: pass`, no quiet defaults when something is missing, no returning `None` where a value is expected.

The **one sanctioned soft-error pattern** is `DemonstrationWrapper.step()`, which catches physics/IK exceptions and surfaces them as `info["status"] = "error"` + `info["error_message"]`. This is an explicit, structured error return — callers must check it.

- Script-level step loops must check `info.get("status") == "error"` and handle it explicitly (see `scripts/run_example.py`).
- Do **not** wrap `env.step()` in a bare `try/except Exception`.
- Do **not** add silent fallback branches or default-return paths unless the user explicitly requests graceful degradation.

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

### Test Categories

| Directory | Marks | Cost | What it covers |
|-----------|-------|------|----------------|
| `tests/lightweight/` | `lightweight` | Seconds | Pure logic: label matching, planners, AST checks, mocks |
| `tests/dataset/` | `dataset`, `gpu` | Minutes | Full env loop: sim + wrapper + HDF5 record/replay |

`tests/dataset/` uses a **session-scoped hash cache** (`_shared/dataset_generation.py`) so identical env/seed/difficulty combinations only generate data once per pytest session.

---

## Testing Philosophy

**Prefer end-to-end, heavy tests.** Tests that spin up real ManiSkill environments, record HDF5 datasets, and replay them are strongly preferred over pure unit tests.

- Do **not** mock the physics engine or the wrapper stack — divergence from real behavior has caused silent bugs before.
- Use mocks only for import-level stubs when ManiSkill/SAPIEN is genuinely unavailable in the current environment.
- When adding new behaviour, write an end-to-end test that: (1) builds a real `BenchmarkEnvBuilder`, (2) records a trajectory, (3) replays it and asserts the full output contract.
- Fall back to lightweight/AST tests only when physics rendering is genuinely unavailable.
- Do not skip or `xfail` without a comment explaining why — prefer fixing the root cause.

### Adding New Tests

- **Pure logic** → `tests/lightweight/test_<topic>.py`, mark `@pytest.mark.lightweight`
- **Sim-dependent** → `tests/dataset/test_<topic>.py`, mark `@pytest.mark.dataset`
- Register any new dataset fixture in `tests/dataset/conftest.py` following the existing `dataset_factory` pattern.

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

doc/                    # supplementary documentation
  env_format.md         # observation/action format spec
  h5_data_format.md     # HDF5 dataset format
  docker_installation.md
```

---

## Instantiating an Environment

```python
from robomme.env_record_wrapper import BenchmarkEnvBuilder

builder = BenchmarkEnvBuilder(
    env_id="PickXtimes",          # task name
    dataset="test",                # "train" | "val" | "test"
    action_space="joint_angle",    # "joint_angle" | "ee_pose" | "waypoint" | "multi_choice"
    gui_render=False,
)
env = builder.make_env_for_episode(episode_idx=0)
obs, info = env.reset()
task_goal = info["task_goal"][0]

obs, reward, terminated, truncated, info = env.step(action)
# info["status"] == "error" means the step raised internally — do not ignore it
```

---

## Task Suites

| Suite      | Focus             | Tasks                                                        |
|------------|-------------------|------------------------------------------------------------|
| Counting   | Temporal memory   | BinFill, PickXtimes, SwingXtimes, StopCube                 |
| Permanence | Spatial memory    | VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap|
| Reference  | Object memory     | PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder|
| Imitation  | Procedural memory | MoveCube, InsertPeg, PatternLock, RouteStick                |

Dataset splits: `train` (100 ep/task), `val` (50 ep/task, fixed seeds), `test` (50 ep/task, held-out).

---

## Challenge Interface (CVPR 2026)

```bash
uv sync --group server

# Terminal 0: policy server
uv run python -m challenge_interface.scripts.deploy --port 8001

# Terminal 1: local evaluation client
uv run python -m challenge_interface.scripts.phase1_eval --port 8001
```

Participants implement `Policy.infer()` and `Policy.reset()` in `challenge_interface/policy.py`.

---

## Headless / GPU Notes

Tasks require a display for Vulkan rendering. For headless machines:

```bash
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

Tests marked `gpu` will fail without a working display or the env vars above. See `doc/docker_installation.md` for more options.
