---
trigger: always_on
---

# Python Environment Requirements (uv)

This project is managed using **uv**. Agents MUST use the uv-managed virtual environment to run Python.

## Hard Requirements

Agents MUST follow these rules:

- MUST use `uv` to create and manage the virtual environment
- MUST run Python using `uv run` or the `.venv/bin/python` interpreter
- MUST NOT use system Python
- MUST NOT use conda
- MUST NOT use pip outside uv
- MUST NOT create alternative virtual environments

Failure to follow these rules will result in incorrect dependencies and runtime errors.

---

## Environment Setup

From the repository root:

```bash
uv venv
uv sync